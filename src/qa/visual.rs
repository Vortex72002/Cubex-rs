use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Mat4, Vec3};
use image::RgbaImage;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::autotest::SURVIVAL_NETHER_SEED;
use crate::config::EngineSettings;
use crate::lua_api::userdata::{ItemStackData, ObjectState};
use crate::paths::EnginePaths;
use crate::physics::collision::{GameMode, PlayerController};
use crate::registry::{EngineRegistries, CONTENT_AIR, RAM_MIC_ITEM_ID};
use crate::render::camera::{compute_camera_rig, CameraMode};
use crate::render::mesher::mesh_chunk_with_atlas;
use crate::render::{ViewportCamera, WgpuRenderer};
use crate::textures::{is_carterpcs_skin, list_available_skins, resolve_skin_rgba, TextureAtlas};
use crate::ui::creative_inventory::{render_creative_inventory, CreativeInventoryState};
use crate::ui::formspec::InventoryCursor;
use crate::ui::hud::render_hud;
use crate::ui::theme::{ItemIconCache, McTheme};
use crate::world::light::propagate_world_light;
use crate::world::mapgen::{
    chunk_y_range_for, find_spawn_pos, generate_chunk_for_world, terrain_height_for, MapgenPalette,
    SEA_LEVEL,
};
use crate::world::nether::{ensure_nether_columns, NetherPalette};
use crate::world::structures::{locate_nearest_structure, StructureKind};
use crate::world::{DimensionId, MobManager, NativeMobKind, WorldGenType, WorldStore};

pub const VISUAL_DELTA_THRESHOLD: f32 = 12.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualFrameResult {
    pub name: String,
    pub output_path: String,
    pub baseline_path: String,
    pub width: u32,
    pub height: u32,
    pub mean_luma: f32,
    pub non_black_ratio: f32,
    pub rgb_stddev: f32,
    pub baseline_status: String,
    pub mean_color_delta: Option<f32>,
    pub changed_pixel_pct: Option<f32>,
    pub passed: bool,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfTelemetry {
    pub avg_fps: f32,
    pub one_percent_low_fps: f32,
    pub benchmark_frames: u32,
    pub chunks_meshed: usize,
    pub avg_chunk_mesh_ms: f64,
    pub peak_chunk_mesh_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualRegressionReport {
    pub passed: bool,
    pub update_baselines: bool,
    pub delta_threshold: f32,
    pub frames: Vec<VisualFrameResult>,
    pub telemetry: PerfTelemetry,
    pub failures: Vec<String>,
}

/// Analyze image brightness/variance (catching blank/solid-black/corrupt frames),
/// save output PNG, and either update golden baseline or compare against `baselines_dir/<name>`.
pub fn analyze_and_compare_frame(
    name: &str,
    img: &RgbaImage,
    out_dir: &Path,
    baselines_dir: &Path,
    update_baselines: bool,
    delta_threshold: f32,
) -> anyhow::Result<VisualFrameResult> {
    let w = img.width();
    let h = img.height();
    let raw = img.as_raw();
    let px_count = (w as usize * h as usize).max(1);

    let mut luma_sum = 0.0f64;
    let mut non_black = 0usize;
    let mut sum_rgb = 0.0f64;
    let mut sum_rgb_sq = 0.0f64;

    for px in raw.chunks_exact(4) {
        let r = px[0] as f64;
        let g = px[1] as f64;
        let b = px[2] as f64;
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        luma_sum += lum;
        if px[0] as u32 + px[1] as u32 + px[2] as u32 > 12 {
            non_black += 1;
        }
        let m = (r + g + b) / 3.0;
        sum_rgb += m;
        sum_rgb_sq += m * m;
    }

    let mean_luma = (luma_sum / px_count as f64) as f32;
    let non_black_ratio = non_black as f32 / px_count as f32;
    let mean_rgb = sum_rgb / px_count as f64;
    let var = (sum_rgb_sq / px_count as f64 - mean_rgb * mean_rgb).max(0.0);
    let rgb_stddev = var.sqrt() as f32;

    let visual_out_dir = out_dir.join("visual");
    std::fs::create_dir_all(&visual_out_dir)?;
    let out_path = visual_out_dir.join(name);
    img.save(&out_path)?;

    let baseline_path = baselines_dir.join(name);
    let mut baseline_status = "no_baseline".to_string();
    let mut mean_color_delta = None;
    let mut changed_pixel_pct = None;
    let mut failure_reason = None;

    if mean_luma < 3.0 || non_black_ratio < 0.05 || rgb_stddev < 2.0 {
        failure_reason = Some(format!(
            "Corrupt/blank frame (mean_luma={mean_luma:.2}, non_black={non_black_ratio:.3}, stddev={rgb_stddev:.2})"
        ));
    } else if update_baselines {
        std::fs::create_dir_all(baselines_dir)?;
        img.save(&baseline_path)?;
        baseline_status = "updated".into();
    } else if baseline_path.is_file() {
        let base = image::open(&baseline_path)?.to_rgba8();
        if base.width() != w || base.height() != h {
            baseline_status = "dimension_mismatch".into();
            failure_reason = Some(format!(
                "Dimension mismatch vs baseline: got {}x{}, expected {}x{}",
                w,
                h,
                base.width(),
                base.height()
            ));
        } else {
            let mut abs_sum = 0.0f64;
            let mut changed = 0usize;
            for (cur_px, base_px) in raw.chunks_exact(4).zip(base.as_raw().chunks_exact(4)) {
                let d = ((cur_px[0] as i32 - base_px[0] as i32).abs()
                    + (cur_px[1] as i32 - base_px[1] as i32).abs()
                    + (cur_px[2] as i32 - base_px[2] as i32).abs()) as f64
                    / 3.0;
                abs_sum += d;
                if d > 8.0 {
                    changed += 1;
                }
            }
            let mean_delta = (abs_sum / px_count as f64) as f32;
            let ch_pct = (changed as f32 / px_count as f32) * 100.0;
            mean_color_delta = Some(mean_delta);
            changed_pixel_pct = Some(ch_pct);
            if mean_delta > delta_threshold {
                baseline_status = "regressed".into();
                failure_reason = Some(format!(
                    "Visual delta {mean_delta:.2} exceeded threshold {delta_threshold:.2} ({ch_pct:.1}% changed pixels)"
                ));
            } else {
                baseline_status = "matched".into();
            }
        }
    }

    Ok(VisualFrameResult {
        name: name.to_string(),
        output_path: out_path.to_string_lossy().to_string(),
        baseline_path: baseline_path.to_string_lossy().to_string(),
        width: w,
        height: h,
        mean_luma,
        non_black_ratio,
        rgb_stddev,
        baseline_status,
        mean_color_delta,
        changed_pixel_pct,
        passed: failure_reason.is_none(),
        failure_reason,
    })
}

fn generate_and_upload_columns(
    world: &WorldStore,
    cols: &[(i32, i32)],
    seed: u64,
    pal: &MapgenPalette,
    reg: &EngineRegistries,
    atlas: &TextureAtlas,
    renderer: &mut WgpuRenderer,
    smooth_lighting: bool,
    mesh_times_ms: &mut Vec<f64>,
) {
    let mut new_cpos = Vec::new();
    for &(cx, cz) in cols {
        for cy in chunk_y_range_for(WorldGenType::Default) {
            let cp = IVec3::new(cx, cy, cz);
            if world.get_chunk(cp).is_none() {
                world.insert_chunk(generate_chunk_for_world(
                    cp,
                    seed,
                    WorldGenType::Default,
                    pal,
                    reg,
                ));
                new_cpos.push(cp);
            }
        }
    }
    if !new_cpos.is_empty() {
        propagate_world_light(world, &new_cpos, reg);
    }
    for &(cx, cz) in cols {
        for cy in chunk_y_range_for(WorldGenType::Default) {
            let cp = IVec3::new(cx, cy, cz);
            if let Some(snap) = world.snapshot_chunk(cp) {
                let t0 = Instant::now();
                let mesh = mesh_chunk_with_atlas(&snap, reg, atlas, smooth_lighting);
                mesh_times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                renderer.upload_chunk_mesh_for_dimension(mesh, DimensionId::Overworld);
            }
        }
    }
}

/// Phase 4: Capture all 5 milestone `wgpu` screenshots offscreen, check against `baselines_dir`,
/// and measure average FPS, 1% low FPS, and peak chunk-meshing latency.
pub fn run_visual_regression(
    paths: &EnginePaths,
    reg: &EngineRegistries,
    atlas: &Arc<TextureAtlas>,
    settings: &EngineSettings,
    out_dir: &Path,
    baselines_dir: &Path,
    update_baselines: bool,
) -> anyhow::Result<VisualRegressionReport> {
    let mut qa_settings = settings.clone();
    qa_settings.render_distance_chunks = qa_settings.render_distance_chunks.max(4);
    qa_settings.smooth_lighting = true;

    let (w, h) = (1280u32, 720u32);
    let mut renderer = WgpuRenderer::new_headless(w, h, atlas)?;
    let egui_ctx = egui::Context::default();
    let theme = McTheme::load(&egui_ctx);
    let mut icon_cache = ItemIconCache::new();
    let mut mesh_times_ms = Vec::new();
    let mut frames = Vec::with_capacity(5);
    let mut failures = Vec::new();

    let tmp_dir = std::env::temp_dir().join("cubex_qa_visual_worlds");
    let _ = std::fs::remove_dir_all(&tmp_dir);

    let ow_pal = MapgenPalette::from_registries_with_game_dir(reg, None);
    let nw_pal = NetherPalette::from_registries(reg);
    let ow = Arc::new(WorldStore::new(&tmp_dir.join("ow_spawn"), SURVIVAL_NETHER_SEED));

    let sp = find_spawn_pos(SURVIVAL_NETHER_SEED, WorldGenType::Default);
    let scx = sp.x.div_euclid(16);
    let scz = sp.z.div_euclid(16);
    let mut spawn_cols = Vec::new();
    for dz in -2..=2 {
        for dx in -2..=2 {
            spawn_cols.push((scx + dx, scz + dz));
        }
    }
    generate_and_upload_columns(
        &ow,
        &spawn_cols,
        SURVIVAL_NETHER_SEED,
        &ow_pal,
        reg,
        atlas,
        &mut renderer,
        qa_settings.smooth_lighting,
        &mut mesh_times_ms,
    );

    let surf_y = terrain_height_for(sp.x, sp.z, SURVIVAL_NETHER_SEED, WorldGenType::Default)
        .max(SEA_LEVEL + 1);
    let mut player = PlayerController::default();
    player.gamemode = GameMode::Survival;
    player.pos = Vec3::new(sp.x as f32 + 0.5, surf_y as f32 + 1.0, sp.z as f32 + 0.5);
    player.yaw = 0.25;
    player.pitch = -0.14;
    let ps = Arc::new(RwLock::new(ObjectState::new_player("VisualQA")));

    // Helper closure for standard egui frame build
    let make_raw_input = |t: f64| egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(w as f32, h as f32),
        )),
        time: Some(t),
        ..Default::default()
    };

    // 1. 01_spawn.png
    let rig_1p = compute_camera_rig(
        CameraMode::FirstPerson,
        &player,
        w as f32 / h as f32,
        &qa_settings,
        &ow,
        reg,
    );
    let hud_out = egui_ctx.run(make_raw_input(0.2), |ctx| {
        render_hud(
            ctx,
            &player,
            &ps,
            &qa_settings,
            reg,
            None,
            60.0,
            16.6,
            spawn_cols.len() * 8,
            40,
            64,
            "HeadlessQA",
            &[],
            None,
        );
    });
    renderer.show_player_model = false;
    renderer.show_viewmodel = true;
    let cam_01 = ViewportCamera {
        cam_pos: rig_1p.eye_pos,
        view_proj: rig_1p.view_proj,
        viewport_px: [0, 0, w, h],
        player_slot: 0,
        draw_local_player_mask: 0,
        draw_viewmodel: false,
        dimension: DimensionId::Overworld,
    };
    let img_01 = renderer.capture_viewport_with_egui_rgba(
        cam_01,
        w,
        h,
        0.34,
        &qa_settings,
        Some(&egui_ctx),
        Some(hud_out),
        1.0,
    )?;
    frames.push(analyze_and_compare_frame(
        "01_spawn.png",
        &img_01,
        out_dir,
        baselines_dir,
        update_baselines,
        VISUAL_DELTA_THRESHOLD,
    )?);

    // 2. 02_carterpcs_3rd_person.png
    let skins = list_available_skins(&paths.skins_dir, &atlas.player_skin);
    let carter_info = skins
        .iter()
        .find(|s| is_carterpcs_skin(&s.id))
        .cloned()
        .unwrap_or_else(|| skins[0].clone());
    let carter_rgba = resolve_skin_rgba(&skins, &carter_info.id, &atlas.player_skin);
    renderer.set_player_skin(0, &carter_info.id, &carter_rgba);
    if reg.get_item_def(RAM_MIC_ITEM_ID).is_some() {
        let psw = ps.write();
        let mut mg = psw.meta.write();
        let main = mg.inventories.entry("main".into()).or_insert_with(|| vec![Default::default(); 32]);
        if !main.is_empty() {
            main[0] = ItemStackData::from_spec(&format!("{RAM_MIC_ITEM_ID} 1"));
        }
    }
    player.yaw = 0.55;
    player.pitch = -0.10;
    renderer.update_player_model_with_swing(
        player.pos,
        player.yaw,
        player.pitch,
        0.0,
        false,
        false,
        0.0,
        Some(RAM_MIC_ITEM_ID),
        0.05,
        15,
        0,
        reg,
        atlas,
    );
    renderer.show_player_model = true;
    renderer.show_viewmodel = false;
    let rig_3p = compute_camera_rig(
        CameraMode::ThirdPersonFront,
        &player,
        w as f32 / h as f32,
        &qa_settings,
        &ow,
        reg,
    );
    player.yaw = 0.25;
    player.pitch = -0.14;
    let cam_02 = ViewportCamera {
        cam_pos: rig_3p.eye_pos,
        view_proj: rig_3p.view_proj,
        viewport_px: [0, 0, w, h],
        player_slot: 0,
        draw_local_player_mask: 1,
        draw_viewmodel: false,
        dimension: DimensionId::Overworld,
    };
    let img_02 = renderer.capture_viewport_with_egui_rgba(
        cam_02,
        w,
        h,
        0.34,
        &qa_settings,
        None,
        None,
        1.0,
    )?;
    frames.push(analyze_and_compare_frame(
        "02_carterpcs_3rd_person.png",
        &img_02,
        out_dir,
        baselines_dir,
        update_baselines,
        VISUAL_DELTA_THRESHOLD,
    )?);

    // 3. 03_creative_catalog.png
    renderer.show_player_model = false;
    let mut creative_state = CreativeInventoryState::default();
    let mut inv_cursor = InventoryCursor::default();
    // Run one layout warm-up frame so egui's Area sizing pass settles before capture,
    // and preserve any font/icon texture deltas uploaded during pass 1.
    let warmup = egui_ctx.run(make_raw_input(0.36), |ctx| {
        let screen = ctx.screen_rect();
        let _ = render_creative_inventory(
            ctx,
            &theme,
            &mut creative_state,
            &ps,
            0,
            reg,
            atlas,
            &mut icon_cache,
            &mut inv_cursor,
            screen,
            7,
            None,
        );
    });
    let mut cat_out = egui_ctx.run(make_raw_input(0.40), |ctx| {
        let screen = ctx.screen_rect();
        let _ = render_creative_inventory(
            ctx,
            &theme,
            &mut creative_state,
            &ps,
            0,
            reg,
            atlas,
            &mut icon_cache,
            &mut inv_cursor,
            screen,
            7,
            None,
        );
    });
    let mut merged_set = warmup.textures_delta.set;
    merged_set.extend(cat_out.textures_delta.set);
    cat_out.textures_delta.set = merged_set;
    let mut merged_free = warmup.textures_delta.free;
    merged_free.extend(cat_out.textures_delta.free);
    cat_out.textures_delta.free = merged_free;

    fn shape_contains_text(shape: &egui::epaint::Shape, needle: &str) -> bool {
        match shape {
            egui::epaint::Shape::Text(t) => t.galley.text().contains(needle),
            egui::epaint::Shape::Vec(v) => v.iter().any(|s| shape_contains_text(s, needle)),
            _ => false,
        }
    }
    let has_creative_title = cat_out
        .shapes
        .iter()
        .any(|s| shape_contains_text(&s.shape, "Creative Catalog"));
    anyhow::ensure!(
        has_creative_title,
        "Creative Catalog UI did not emit visible 'Creative Catalog' header shapes"
    );
    let cam_03 = ViewportCamera {
        cam_pos: rig_1p.eye_pos,
        view_proj: rig_1p.view_proj,
        viewport_px: [0, 0, w, h],
        player_slot: 0,
        draw_local_player_mask: 0,
        draw_viewmodel: false,
        dimension: DimensionId::Overworld,
    };
    let img_03 = renderer.capture_viewport_with_egui_rgba(
        cam_03,
        w,
        h,
        0.34,
        &qa_settings,
        Some(&egui_ctx),
        Some(cat_out),
        1.0,
    )?;
    frames.push(analyze_and_compare_frame(
        "03_creative_catalog.png",
        &img_03,
        out_dir,
        baselines_dir,
        update_baselines,
        VISUAL_DELTA_THRESHOLD,
    )?);

    // 4. 04_nether_fortress.png
    let nw = Arc::new(WorldStore::new_for_dimension(
        &tmp_dir.join("nw_fort"),
        SURVIVAL_NETHER_SEED,
        DimensionId::Nether,
        WorldGenType::Default,
        crate::world::Difficulty::Normal,
        crate::world::Gamerules::default(),
        "Survival",
    ));
    let fort = locate_nearest_structure(
        StructureKind::NetherFortress,
        SURVIVAL_NETHER_SEED,
        WorldGenType::Default,
        (0, 0),
    )
    .expect("Nether Fortress anchor");
    let ncx = fort.origin.x.div_euclid(16);
    let ncz = fort.origin.z.div_euclid(16);
    let n_chunks = ensure_nether_columns(&nw, ncx, ncz, 2, SURVIVAL_NETHER_SEED, &nw_pal, reg);

    let fort_cam_pos = fort.teleport_pos + Vec3::new(0.0, 2.2, -5.5);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let p = IVec3::new(
                    fort_cam_pos.x.floor() as i32 + dx,
                    fort_cam_pos.y.floor() as i32 + dy,
                    fort_cam_pos.z.floor() as i32 + dz,
                );
                nw.set_node(p, CONTENT_AIR, 15, 0);
            }
        }
    }
    propagate_world_light(&nw, &n_chunks, reg);
    for cp in &n_chunks {
        if let Some(snap) = nw.snapshot_chunk(*cp) {
            let t0 = Instant::now();
            let mesh = mesh_chunk_with_atlas(&snap, reg, atlas, qa_settings.smooth_lighting);
            mesh_times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            renderer.upload_chunk_mesh_for_dimension(mesh, DimensionId::Nether);
        }
    }
    let mut n_mobs = MobManager::new(SURVIVAL_NETHER_SEED);
    n_mobs.spawn_mob(
        NativeMobKind::Blaze,
        DimensionId::Nether,
        fort.teleport_pos + Vec3::new(-1.2, 0.2, 1.5),
        1,
    );
    n_mobs.spawn_mob(
        NativeMobKind::WitherSkeleton,
        DimensionId::Nether,
        fort.teleport_pos + Vec3::new(1.5, 0.2, 2.0),
        1,
    );
    let n_snaps = n_mobs.render_snapshots_for(DimensionId::Nether, &nw);
    renderer.update_nether_entities(&n_snaps, reg, atlas);

    let fort_proj = Mat4::perspective_rh(72.0f32.to_radians(), w as f32 / h as f32, 0.1, 128.0)
        * Mat4::look_at_rh(fort_cam_pos, fort.teleport_pos + Vec3::new(0.0, 1.0, 2.0), Vec3::Y);
    let cam_04 = ViewportCamera {
        cam_pos: fort_cam_pos,
        view_proj: fort_proj,
        viewport_px: [0, 0, w, h],
        player_slot: 0,
        draw_local_player_mask: 0,
        draw_viewmodel: false,
        dimension: DimensionId::Nether,
    };
    let img_04 = renderer.capture_viewport_with_egui_rgba(
        cam_04,
        w,
        h,
        0.5,
        &qa_settings,
        None,
        None,
        1.0,
    )?;
    frames.push(analyze_and_compare_frame(
        "04_nether_fortress.png",
        &img_04,
        out_dir,
        baselines_dir,
        update_baselines,
        VISUAL_DELTA_THRESHOLD,
    )?);

    // 5. 05_ocean_monument.png
    renderer.prune_distant_chunks_for_dimension(DimensionId::Overworld, &[], 0);
    let ow_mon = Arc::new(WorldStore::new(&tmp_dir.join("ow_mon"), SURVIVAL_NETHER_SEED));
    let mon = locate_nearest_structure(
        StructureKind::OceanMonument,
        SURVIVAL_NETHER_SEED,
        WorldGenType::Default,
        (0, 0),
    )
    .expect("Ocean Monument anchor");
    let mcx = mon.origin.x.div_euclid(16);
    let mcz = mon.origin.z.div_euclid(16);
    let mut mon_cols = Vec::new();
    for dz in -3..=3 {
        for dx in -3..=3 {
            mon_cols.push((mcx + dx, mcz + dz));
        }
    }
    generate_and_upload_columns(
        &ow_mon,
        &mon_cols,
        SURVIVAL_NETHER_SEED,
        &ow_pal,
        reg,
        atlas,
        &mut renderer,
        qa_settings.smooth_lighting,
        &mut mesh_times_ms,
    );
    let mut mon_mobs = MobManager::new(SURVIVAL_NETHER_SEED ^ 77);
    mon_mobs.spawn_mob(
        NativeMobKind::Guardian,
        DimensionId::Overworld,
        Vec3::new(mon.origin.x as f32 + 2.0, (SEA_LEVEL + 2) as f32, mon.origin.z as f32 + 2.0),
        1,
    );
    renderer.update_entities(&mon_mobs.render_snapshots_for(DimensionId::Overworld, &ow_mon), reg, atlas);

    let mon_cam_pos = Vec3::new(
        mon.origin.x as f32 - 22.0,
        (mon.bbox_max.y as f32 + 12.0).max((SEA_LEVEL + 14) as f32),
        mon.origin.z as f32 - 22.0,
    );
    let mon_target = Vec3::new(
        mon.origin.x as f32,
        (mon.origin.y as f32 + 4.0).min(SEA_LEVEL as f32),
        mon.origin.z as f32,
    );
    let mon_proj = Mat4::perspective_rh(72.0f32.to_radians(), w as f32 / h as f32, 0.1, 160.0)
        * Mat4::look_at_rh(mon_cam_pos, mon_target, Vec3::Y);
    let cam_05 = ViewportCamera {
        cam_pos: mon_cam_pos,
        view_proj: mon_proj,
        viewport_px: [0, 0, w, h],
        player_slot: 0,
        draw_local_player_mask: 0,
        draw_viewmodel: false,
        dimension: DimensionId::Overworld,
    };
    let img_05 = renderer.capture_viewport_with_egui_rgba(
        cam_05,
        w,
        h,
        0.34,
        &qa_settings,
        None,
        None,
        1.0,
    )?;
    frames.push(analyze_and_compare_frame(
        "05_ocean_monument.png",
        &img_05,
        out_dir,
        baselines_dir,
        update_baselines,
        VISUAL_DELTA_THRESHOLD,
    )?);

    // Performance benchmark burst (24 frames on offscreen target matching normal render path)
    let bench_frames = 24u32;
    let mut frame_ms_list = Vec::with_capacity(bench_frames as usize);
    for _ in 0..bench_frames {
        let t0 = Instant::now();
        renderer.render_frame_in_dimension(
            mon_cam_pos,
            mon_proj,
            DimensionId::Overworld,
            0.34,
            &qa_settings,
            None,
            None,
            1.0,
        )?;
        renderer.wait_idle();
        frame_ms_list.push((t0.elapsed().as_secs_f64() * 1000.0).max(0.05));
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);

    for fr in &frames {
        if !fr.passed {
            if let Some(ref r) = fr.failure_reason {
                failures.push(format!("{}: {r}", fr.name));
            }
        }
    }

    let sum_ms: f64 = frame_ms_list.iter().sum();
    let avg_frame_ms = (sum_ms / bench_frames as f64).max(0.01);
    let avg_fps = (1000.0 / avg_frame_ms) as f32;
    let mut sorted_ms = frame_ms_list;
    sorted_ms.sort_by(|a, b| a.total_cmp(b));
    let p99_ms = sorted_ms[((sorted_ms.len() as f32 * 0.99) as usize).min(sorted_ms.len() - 1)];
    let one_percent_low_fps = (1000.0 / p99_ms.max(0.01)) as f32;

    let chunks_meshed = mesh_times_ms.len();
    let avg_chunk_mesh_ms = if chunks_meshed > 0 {
        mesh_times_ms.iter().sum::<f64>() / chunks_meshed as f64
    } else {
        0.0
    };
    let peak_chunk_mesh_ms = mesh_times_ms
        .iter()
        .copied()
        .fold(0.0f64, f64::max);

    Ok(VisualRegressionReport {
        passed: failures.is_empty(),
        update_baselines,
        delta_threshold: VISUAL_DELTA_THRESHOLD,
        frames,
        telemetry: PerfTelemetry {
            avg_fps,
            one_percent_low_fps,
            benchmark_frames: bench_frames,
            chunks_meshed,
            avg_chunk_mesh_ms,
            peak_chunk_mesh_ms,
        },
        failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn visual_compare_rejects_black_and_detects_regressions() {
        let dir = std::env::temp_dir().join("cubex_qa_visual_unit_test");
        let _ = std::fs::remove_dir_all(&dir);
        let out_dir = dir.join("out");
        let base_dir = dir.join("baselines");

        let black = RgbaImage::from_pixel(64, 64, Rgba([0, 0, 0, 255]));
        let res_black =
            analyze_and_compare_frame("test.png", &black, &out_dir, &base_dir, true, 12.0).unwrap();
        assert!(!res_black.passed);

        let pattern = RgbaImage::from_fn(64, 64, |x, y| {
            Rgba([((x * 4) % 255) as u8, ((y * 4) % 255) as u8, 140, 255])
        });
        let res_upd =
            analyze_and_compare_frame("good.png", &pattern, &out_dir, &base_dir, true, 12.0)
                .unwrap();
        assert!(res_upd.passed);
        assert_eq!(res_upd.baseline_status, "updated");

        let res_match =
            analyze_and_compare_frame("good.png", &pattern, &out_dir, &base_dir, false, 12.0)
                .unwrap();
        assert!(res_match.passed);
        assert_eq!(res_match.baseline_status, "matched");
        assert!(res_match.mean_color_delta.unwrap() < 0.01);

        let inverted = RgbaImage::from_fn(64, 64, |x, _y| {
            Rgba([255 - ((x * 4) % 255) as u8, 10, 240, 255])
        });
        let res_reg =
            analyze_and_compare_frame("good.png", &inverted, &out_dir, &base_dir, false, 12.0)
                .unwrap();
        assert!(!res_reg.passed);
        assert_eq!(res_reg.baseline_status, "regressed");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
