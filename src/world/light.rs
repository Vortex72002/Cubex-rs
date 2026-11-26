use glam::IVec3;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_IGNORE};
use crate::world::{local_index, world_to_chunk, Chunk, WorldStore, CHUNK_SIZE};

pub const LIGHT_SUN: u8 = 15;
pub const LIGHT_MAX: u8 = 14;

const DIRS_I: [(isize, isize, isize); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

const DIRS_V: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

#[inline(always)]
pub fn get_sky_light(param1: u8) -> u8 {
    param1 & 0x0F
}

#[inline(always)]
pub fn get_block_light(param1: u8) -> u8 {
    (param1 >> 4) & 0x0F
}

#[inline(always)]
pub fn pack_light(sky: u8, block: u8) -> u8 {
    ((block.min(LIGHT_SUN)) << 4) | (sky.min(LIGHT_SUN))
}

/// Luanti perceptual light LUT curve (`src/light.cpp`), mapping 0..15 to [0.06, 1.0].
pub fn light_lut_factor(level: u8) -> f32 {
    if level >= LIGHT_SUN {
        return 1.0;
    }
    if level == 0 {
        return 0.06;
    }
    let x = level as f32 / LIGHT_SUN as f32;
    // Luanti default curve: alpha=0.25, beta=1.35 -> smooth concave-up brightness response
    let alpha = 0.25f32;
    let beta = 1.35f32;
    let a = alpha + beta - 2.0;
    let b = 3.0 - 2.0 * alpha - beta;
    let c = alpha;
    let brightness = ((a * x + b) * x + c) * x;
    brightness.clamp(0.06, 1.0)
}

#[inline]
pub fn is_light_propagating(id: u16, registries: &EngineRegistries) -> bool {
    if id == CONTENT_AIR {
        return true;
    }
    if id == CONTENT_IGNORE {
        return false;
    }
    registries
        .get_node_def(id)
        .map(|d| d.sunlight_propagates || d.paramtype == "light" || d.drawtype.is_transparent())
        .unwrap_or(false)
}

#[inline]
pub fn is_sunlight_propagating(id: u16, registries: &EngineRegistries) -> bool {
    if id == CONTENT_AIR {
        return true;
    }
    if id == CONTENT_IGNORE {
        return false;
    }
    registries
        .get_node_def(id)
        .map(|d| d.sunlight_propagates)
        .unwrap_or(false)
}

#[inline]
pub fn is_foliage_or_soft_light(id: u16, registries: &EngineRegistries) -> bool {
    if id == CONTENT_AIR || id == CONTENT_IGNORE {
        return false;
    }
    registries
        .get_node_def(id)
        .map(|d| {
            matches!(
                d.drawtype,
                crate::registry::DrawType::Allfaces
                    | crate::registry::DrawType::AllfacesOptional
                    | crate::registry::DrawType::Plantlike
            ) || d.groups.get("leaves").copied().unwrap_or(0) > 0
        })
        .unwrap_or(false)
}

/// Vertical skylight transmission step when sunlight passes downward through `id`.
/// Open/sunlit nodes pass light unattenuated; leaves/foliage attenuate gently (`15 -> 14 -> 13`)
/// so canopies and trunk tops remain sunlit instead of collapsing to 0; liquids attenuate by 2;
/// opaque solid nodes block vertical skylight (`0`).
#[inline]
pub fn vertical_sky_step(sky: u8, id: u16, registries: &EngineRegistries) -> u8 {
    if sky == 0 || id == CONTENT_IGNORE {
        return 0;
    }
    if is_sunlight_propagating(id, registries) {
        return sky;
    }
    if is_foliage_or_soft_light(id, registries) {
        return sky.saturating_sub(1);
    }
    if registries
        .get_node_def(id)
        .map(|d| {
            matches!(
                d.drawtype,
                crate::registry::DrawType::Liquid | crate::registry::DrawType::FlowingLiquid
            )
        })
        .unwrap_or(false)
    {
        return sky.saturating_sub(2);
    }
    0
}

#[inline]
pub fn node_light_source(id: u16, registries: &EngineRegistries) -> u8 {
    if id == CONTENT_AIR || id == CONTENT_IGNORE {
        return 0;
    }
    registries
        .get_node_def(id)
        .map(|d| d.light_source.min(LIGHT_SUN))
        .unwrap_or(0)
}

/// Compute dual-channel skylight (0..15) + blocklight (0..15) within a single chunk.
/// Ports Luanti's vertical sky column pass + 6-neighbor BFS flood-fill for both light banks.
pub fn propagate_chunk_light(chunk: &mut Chunk, registries: &EngineRegistries, is_sky_chunk: bool) {
    let mut sky_bfs: VecDeque<(usize, usize, usize, u8)> = VecDeque::new();
    let mut block_bfs: VecDeque<(usize, usize, usize, u8)> = VecDeque::new();

    let start_sky: u8 = if is_sky_chunk { LIGHT_SUN } else { 0 };

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let mut sky = start_sky;
            for y in (0..CHUNK_SIZE).rev() {
                let idx = local_index(x, y, z);
                let id = chunk.nodes[idx];
                let emit = node_light_source(id, registries);

                sky = vertical_sky_step(sky, id, registries);
                let stored_sky = if sky > 0 && is_light_propagating(id, registries) {
                    sky
                } else {
                    0
                };
                chunk.param1[idx] = pack_light(stored_sky, emit);
                if stored_sky > 1 {
                    sky_bfs.push_back((x, y, z, stored_sky));
                }
                if emit > 0 {
                    block_bfs.push_back((x, y, z, emit));
                }
            }
        }
    }

    // Skylight 6-neighbor BFS (propagates into leaves, water, overhangs, and caves)
    while let Some((x, y, z, level)) = sky_bfs.pop_front() {
        if level <= 1 {
            continue;
        }
        let next_lvl = level - 1;
        for (dx, dy, dz) in DIRS_I {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            let nz = z as isize + dz;
            if !(0..16).contains(&nx) || !(0..16).contains(&ny) || !(0..16).contains(&nz) {
                continue;
            }
            let nidx = local_index(nx as usize, ny as usize, nz as usize);
            let nid = chunk.nodes[nidx];
            if !is_light_propagating(nid, registries) {
                continue;
            }
            let cur_sky = get_sky_light(chunk.param1[nidx]);
            if cur_sky < next_lvl {
                chunk.param1[nidx] = (chunk.param1[nidx] & 0xF0) | next_lvl;
                sky_bfs.push_back((nx as usize, ny as usize, nz as usize, next_lvl));
            }
        }
    }

    // Artificial blocklight 6-neighbor BFS
    while let Some((x, y, z, level)) = block_bfs.pop_front() {
        if level <= 1 {
            continue;
        }
        let next_lvl = level - 1;
        for (dx, dy, dz) in DIRS_I {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            let nz = z as isize + dz;
            if !(0..16).contains(&nx) || !(0..16).contains(&ny) || !(0..16).contains(&nz) {
                continue;
            }
            let nidx = local_index(nx as usize, ny as usize, nz as usize);
            let nid = chunk.nodes[nidx];
            if !is_light_propagating(nid, registries) {
                continue;
            }
            let cur_block = get_block_light(chunk.param1[nidx]);
            if cur_block < next_lvl {
                chunk.param1[nidx] = (next_lvl << 4) | (chunk.param1[nidx] & 0x0F);
                block_bfs.push_back((nx as usize, ny as usize, nz as usize, next_lvl));
            }
        }
    }
}

/// Propagate cross-chunk sunlight columns and 6-neighbor BFS sky+block light across a batch
/// of loaded chunks (useful after spawning trees/schematics or loading new chunk columns).
pub fn propagate_world_light(
    world: &WorldStore,
    chunk_coords: &[IVec3],
    registries: &EngineRegistries,
) {
    let _ = propagate_world_light_inner(world, chunk_coords, registries);
}

/// Relight an entire loaded vertical `(cx, cz)` column from open sky down through all loaded Y chunks
/// (`cy = max_cy` down to `min_cy`) and return all chunks whose light or neighbor border changed.
pub fn relight_world_column(
    world: &WorldStore,
    cx: i32,
    cz: i32,
    registries: &EngineRegistries,
) -> Vec<IVec3> {
    let col_coords: Vec<IVec3> = {
        let map = world.chunks.read();
        map.keys()
            .copied()
            .filter(|p| p.x == cx && p.z == cz)
            .collect()
    };
    if col_coords.is_empty() {
        return Vec::new();
    }
    propagate_world_light_inner(world, &col_coords, registries)
        .into_iter()
        .collect()
}

fn propagate_world_light_inner(
    world: &WorldStore,
    chunk_coords: &[IVec3],
    registries: &EngineRegistries,
) -> HashSet<IVec3> {
    if chunk_coords.is_empty() {
        return HashSet::new();
    }

    // Collect all unique (cx, cz) columns and expand each to all loaded Y chunks in that column
    let unique_cols: HashSet<(i32, i32)> = chunk_coords.iter().map(|p| (p.x, p.z)).collect();
    let mut columns: HashMap<(i32, i32), Vec<i32>> = HashMap::new();
    {
        let map = world.chunks.read();
        for &cpos in map.keys() {
            if unique_cols.contains(&(cpos.x, cpos.z)) {
                columns.entry((cpos.x, cpos.z)).or_default().push(cpos.y);
            }
        }
    }
    for &cpos in chunk_coords {
        columns.entry((cpos.x, cpos.z)).or_default().push(cpos.y);
    }

    let mut sky_bfs: VecDeque<(IVec3, u8)> = VecDeque::new();
    let mut block_bfs: VecDeque<(IVec3, u8)> = VecDeque::new();
    let mut touched: HashSet<IVec3> = HashSet::new();
    let mut visited_chunks: HashSet<IVec3> = HashSet::new();

    for ((cx, cz), mut ys) in columns {
        ys.sort_unstable();
        ys.dedup();
        if ys.is_empty() {
            continue;
        }
        let min_cy = *ys.first().unwrap();
        let max_cy = *ys.last().unwrap();
        for &cy in &ys {
            visited_chunks.insert(IVec3::new(cx, cy, cz));
        }

        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wx = cx * 16 + lx as i32;
                let wz = cz * 16 + lz as i32;
                let above_pos = IVec3::new(wx, (max_cy + 1) * 16, wz);
                let (above_id, above_p1, _) = world.get_node(above_pos);
                let mut sky = if above_id == CONTENT_IGNORE {
                    if max_cy >= -2 { LIGHT_SUN } else { 0 }
                } else {
                    get_sky_light(above_p1)
                };

                for cy in (min_cy..=max_cy).rev() {
                    let cpos = IVec3::new(cx, cy, cz);
                    let Some(ch_arc) = world.get_chunk(cpos) else {
                        sky = 0;
                        continue;
                    };
                    let mut ch = ch_arc.write();
                    for ly in (0..CHUNK_SIZE).rev() {
                        let idx = local_index(lx, ly, lz);
                        let id = ch.nodes[idx];
                        let emit = node_light_source(id, registries);
                        let wy = cy * 16 + ly as i32;
                        let wpos = IVec3::new(wx, wy, wz);

                        sky = vertical_sky_step(sky, id, registries);
                        let new_sky = if sky > 0 && is_light_propagating(id, registries) {
                            sky
                        } else {
                            0
                        };
                        let cur_block = get_block_light(ch.param1[idx]);
                        let new_block = cur_block.max(emit);
                        let packed = pack_light(new_sky, new_block);
                        if ch.param1[idx] != packed {
                            ch.param1[idx] = packed;
                            ch.dirty_mesh = true;
                            touched.insert(cpos);
                        }
                        if new_sky > 1 && is_light_propagating(id, registries) {
                            sky_bfs.push_back((wpos, new_sky));
                        }
                        if new_block > 1 {
                            block_bfs.push_back((wpos, new_block));
                        }
                    }
                }
            }
        }
    }

    // Seed boundary neighbors from already-lit surrounding chunks outside visited columns
    for &cpos in &visited_chunks {
        for delta in DIRS_V {
            let np = cpos + delta;
            if visited_chunks.contains(&np) {
                continue;
            }
            let Some(nch_arc) = world.get_chunk(np) else {
                continue;
            };
            let nch = nch_arc.read();
            let base = np * 16;
            for ly in 0..16usize {
                for lz in 0..16usize {
                    for lx in 0..16usize {
                        if lx != 0 && lx != 15 && ly != 0 && ly != 15 && lz != 0 && lz != 15 {
                            continue;
                        }
                        let (id, p1, _) = nch.get(lx, ly, lz);
                        let s = get_sky_light(p1);
                        let b = get_block_light(p1);
                        let wp = base + IVec3::new(lx as i32, ly as i32, lz as i32);
                        if s > 1 && is_light_propagating(id, registries) {
                            sky_bfs.push_back((wp, s));
                        }
                        if b > 1 && (is_light_propagating(id, registries) || node_light_source(id, registries) > 0) {
                            block_bfs.push_back((wp, b));
                        }
                    }
                }
            }
        }
    }

    spread_channel_bfs(world, registries, &mut sky_bfs, false, &mut touched);
    spread_channel_bfs(world, registries, &mut block_bfs, true, &mut touched);

    if !touched.is_empty() {
        let mut dq = world.dirty_queue.write();
        for &cpos in &touched {
            dq.insert(cpos);
        }
    }
    touched
}

#[derive(Copy, Clone)]
enum LightChannel {
    Sky,
    Block,
}

#[inline]
fn get_world_channel(world: &WorldStore, pos: IVec3, ch: LightChannel) -> Option<(u16, u8)> {
    let (cpos, lx, ly, lz) = world_to_chunk(pos);
    let chunk = world.get_chunk(cpos)?;
    let (id, p1, _) = chunk.read().get(lx, ly, lz);
    if id == CONTENT_IGNORE {
        return None;
    }
    let lvl = match ch {
        LightChannel::Sky => get_sky_light(p1),
        LightChannel::Block => get_block_light(p1),
    };
    Some((id, lvl))
}

#[inline]
fn set_world_channel(
    world: &WorldStore,
    pos: IVec3,
    ch: LightChannel,
    val: u8,
    modified_chunks: &mut HashSet<IVec3>,
) {
    let (cpos, lx, ly, lz) = world_to_chunk(pos);
    if let Some(chunk) = world.get_chunk(cpos) {
        let mut g = chunk.write();
        let idx = local_index(lx, ly, lz);
        let old = g.param1[idx];
        let next = match ch {
            LightChannel::Sky => (old & 0xF0) | (val & 0x0F),
            LightChannel::Block => ((val & 0x0F) << 4) | (old & 0x0F),
        };
        if old != next {
            g.param1[idx] = next;
            g.dirty_mesh = true;
            modified_chunks.insert(cpos);
            if lx == 0 {
                modified_chunks.insert(cpos - IVec3::X);
            } else if lx == 15 {
                modified_chunks.insert(cpos + IVec3::X);
            }
            if ly == 0 {
                modified_chunks.insert(cpos - IVec3::Y);
            } else if ly == 15 {
                modified_chunks.insert(cpos + IVec3::Y);
            }
            if lz == 0 {
                modified_chunks.insert(cpos - IVec3::Z);
            } else if lz == 15 {
                modified_chunks.insert(cpos + IVec3::Z);
            }
        }
    }
}

fn spread_channel_bfs(
    world: &WorldStore,
    registries: &EngineRegistries,
    queue: &mut VecDeque<(IVec3, u8)>,
    is_block: bool,
    modified_chunks: &mut HashSet<IVec3>,
) {
    let ch = if is_block {
        LightChannel::Block
    } else {
        LightChannel::Sky
    };
    while let Some((p, level)) = queue.pop_front() {
        if level <= 1 {
            continue;
        }
        let next_lvl = level - 1;
        for d in DIRS_V {
            let np = p + d;
            let Some((nid, ncur)) = get_world_channel(world, np, ch) else {
                continue;
            };
            if !is_light_propagating(nid, registries) {
                continue;
            }
            if ncur < next_lvl {
                set_world_channel(world, np, ch, next_lvl, modified_chunks);
                queue.push_back((np, next_lvl));
            }
        }
    }
}

fn is_direct_sunlit_above(world: &WorldStore, pos: IVec3, registries: &EngineRegistries) -> bool {
    let above = pos + IVec3::Y;
    match get_world_channel(world, above, LightChannel::Sky) {
        Some((aid, alvl)) => alvl == LIGHT_SUN && is_sunlight_propagating(aid, registries),
        None => true, // Above top loaded chunk is open sky
    }
}

/// Incrementally relight both sky and block channels around `pos` after a node change (place/dig).
/// Ports Luanti's unlight-BFS + relight-BFS across chunks (`voxelalgorithms.cpp`).
/// Returns all chunk coordinates whose lighting (or border mesh neighbors) changed.
pub fn relight_around_pos(
    world: &WorldStore,
    pos: IVec3,
    registries: &EngineRegistries,
) -> Vec<IVec3> {
    let mut modified_chunks = HashSet::new();

    // 1. Update Sky channel (including vertical sunlight column directly below `pos`)
    {
        let mut unlight: [Vec<IVec3>; 16] = Default::default();
        let mut relight: VecDeque<(IVec3, u8)> = VecDeque::new();
        let mut unlighted_set: HashSet<IVec3> = HashSet::new();

        // Trace vertical sky column starting at `pos` downward through loaded chunks
        let mut cur_sky = match get_world_channel(world, pos + IVec3::Y, LightChannel::Sky) {
            Some((_, alvl)) => alvl,
            None => LIGHT_SUN,
        };
        let mut y = pos.y;
        loop {
            let wp = IVec3::new(pos.x, y, pos.z);
            let Some((id, old_sky)) = get_world_channel(world, wp, LightChannel::Sky) else {
                break;
            };
            cur_sky = vertical_sky_step(cur_sky, id, registries);
            let target_sky = if cur_sky > 0 && is_light_propagating(id, registries) {
                cur_sky
            } else {
                0
            };
            if target_sky > 0 {
                if old_sky != target_sky {
                    set_world_channel(world, wp, LightChannel::Sky, target_sky, &mut modified_chunks);
                }
                if target_sky > 1 {
                    relight.push_back((wp, target_sky));
                }
                y -= 1;
            } else if y == pos.y || old_sky > 0 {
                if old_sky > 0 {
                    unlight[old_sky as usize].push(wp);
                    unlighted_set.insert(wp);
                }
                set_world_channel(world, wp, LightChannel::Sky, 0, &mut modified_chunks);
                y -= 1;
            } else {
                break;
            }
        }

        // Inspect immediate neighbors of `pos` in case `pos` previously conducted sky or blocked it
        for d in DIRS_V {
            let np = pos + d;
            if unlighted_set.contains(&np) {
                continue;
            }
            if let Some((nid, old_sky)) = get_world_channel(world, np, LightChannel::Sky) {
                if old_sky > 0
                    && !(old_sky == LIGHT_SUN
                        && is_sunlight_propagating(nid, registries)
                        && is_direct_sunlit_above(world, np, registries))
                {
                    unlight[old_sky as usize].push(np);
                    unlighted_set.insert(np);
                    set_world_channel(world, np, LightChannel::Sky, 0, &mut modified_chunks);
                } else if old_sky > 1 && is_light_propagating(nid, registries) {
                    relight.push_back((np, old_sky));
                }
            }
        }

        // Unlight BFS from brightest to dimmest
        for lvl in (1..=15usize).rev() {
            while let Some(p) = unlight[lvl].pop() {
                for d in DIRS_V {
                    let np = p + d;
                    let Some((nid, nsky)) = get_world_channel(world, np, LightChannel::Sky) else {
                        continue;
                    };
                    if nsky == 0 || !is_light_propagating(nid, registries) {
                        continue;
                    }
                    let direct_sun = nsky == LIGHT_SUN
                        && is_sunlight_propagating(nid, registries)
                        && is_direct_sunlit_above(world, np, registries);
                    if !direct_sun && (nsky as usize) < lvl {
                        set_world_channel(world, np, LightChannel::Sky, 0, &mut modified_chunks);
                        unlight[nsky as usize].push(np);
                        unlighted_set.insert(np);
                    } else if (nsky as usize) >= lvl || direct_sun {
                        relight.push_back((np, nsky));
                    }
                }
            }
        }

        // Re-seed from exterior perimeter of unlighted region & vertical sky above unlighted nodes
        for &up in &unlighted_set {
            if let Some((uid, _)) = get_world_channel(world, up, LightChannel::Sky) {
                if is_light_propagating(uid, registries) {
                    let above_sky = match get_world_channel(world, up + IVec3::Y, LightChannel::Sky) {
                        None => vertical_sky_step(LIGHT_SUN, uid, registries),
                        Some((_, alvl)) if !unlighted_set.contains(&(up + IVec3::Y)) => {
                            vertical_sky_step(alvl, uid, registries)
                        }
                        _ => 0,
                    };
                    if above_sky > 0 {
                        set_world_channel(world, up, LightChannel::Sky, above_sky, &mut modified_chunks);
                        if above_sky > 1 {
                            relight.push_back((up, above_sky));
                        }
                    }
                }
            }
            for d in DIRS_V {
                let np = up + d;
                if unlighted_set.contains(&np) {
                    continue;
                }
                if let Some((nid, nsky)) = get_world_channel(world, np, LightChannel::Sky) {
                    if nsky > 1 && is_light_propagating(nid, registries) {
                        relight.push_back((np, nsky));
                    }
                }
            }
        }

        if let Some((pid, psky)) = get_world_channel(world, pos, LightChannel::Sky) {
            if is_light_propagating(pid, registries) {
                if psky > 1 {
                    relight.push_back((pos, psky));
                }
                for d in DIRS_V {
                    if let Some((nid, nsky)) = get_world_channel(world, pos + d, LightChannel::Sky) {
                        if nsky > 1 && is_light_propagating(nid, registries) {
                            relight.push_back((pos + d, nsky));
                        }
                    }
                }
            }
        }

        spread_channel_bfs(world, registries, &mut relight, false, &mut modified_chunks);
    }

    // 2. Update Artificial Block channel (torches, lava, glowstone, etc.)
    {
        let mut unlight: [Vec<IVec3>; 16] = Default::default();
        let mut relight: VecDeque<(IVec3, u8)> = VecDeque::new();
        let mut unlighted_set: HashSet<IVec3> = HashSet::new();

        let (pid, old_block_at_pos) = get_world_channel(world, pos, LightChannel::Block)
            .unwrap_or((CONTENT_IGNORE, 0));
        let emit_at_pos = node_light_source(pid, registries);
        if emit_at_pos > 0 {
            set_world_channel(world, pos, LightChannel::Block, emit_at_pos, &mut modified_chunks);
            relight.push_back((pos, emit_at_pos));
        } else {
            if old_block_at_pos > 0 {
                unlight[old_block_at_pos as usize].push(pos);
            }
            set_world_channel(world, pos, LightChannel::Block, 0, &mut modified_chunks);
            unlighted_set.insert(pos);
        }

        // Inspect 6 neighbors (handles case where `set_node` already cleared `param1` at `pos`!)
        for d in DIRS_V {
            let np = pos + d;
            if let Some((nid, nblk)) = get_world_channel(world, np, LightChannel::Block) {
                let nemit = node_light_source(nid, registries);
                if nemit > 0 {
                    if nblk < nemit {
                        set_world_channel(world, np, LightChannel::Block, nemit, &mut modified_chunks);
                    }
                    relight.push_back((np, nblk.max(nemit)));
                } else if nblk > 0 && is_light_propagating(nid, registries) {
                    unlight[nblk as usize].push(np);
                    unlighted_set.insert(np);
                    set_world_channel(world, np, LightChannel::Block, 0, &mut modified_chunks);
                }
            }
        }

        for lvl in (1..=15usize).rev() {
            while let Some(p) = unlight[lvl].pop() {
                for d in DIRS_V {
                    let np = p + d;
                    let Some((nid, nblk)) = get_world_channel(world, np, LightChannel::Block) else {
                        continue;
                    };
                    let nemit = node_light_source(nid, registries);
                    if nemit > 0 {
                        relight.push_back((np, nblk.max(nemit)));
                        continue;
                    }
                    if nblk == 0 || !is_light_propagating(nid, registries) {
                        continue;
                    }
                    if (nblk as usize) < lvl {
                        set_world_channel(world, np, LightChannel::Block, 0, &mut modified_chunks);
                        unlight[nblk as usize].push(np);
                        unlighted_set.insert(np);
                    } else if (nblk as usize) >= lvl {
                        relight.push_back((np, nblk));
                    }
                }
            }
        }

        // Check exterior boundary of unlighted nodes for surviving light emitters
        for &up in &unlighted_set {
            if let Some((uid, _)) = get_world_channel(world, up, LightChannel::Block) {
                let uemit = node_light_source(uid, registries);
                if uemit > 0 {
                    set_world_channel(world, up, LightChannel::Block, uemit, &mut modified_chunks);
                    relight.push_back((up, uemit));
                }
            }
            for d in DIRS_V {
                let np = up + d;
                if let Some((nid, nblk)) = get_world_channel(world, np, LightChannel::Block) {
                    let nemit = node_light_source(nid, registries);
                    let lvl = nblk.max(nemit);
                    if lvl > 1 && (is_light_propagating(nid, registries) || nemit > 0) {
                        relight.push_back((np, lvl));
                    }
                }
            }
        }

        spread_channel_bfs(world, registries, &mut relight, true, &mut modified_chunks);
    }

    modified_chunks.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, NodeDef};

    #[test]
    fn dual_channel_sky_and_torch_relight_roundtrip() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let leaves = reg.register_node(NodeDef {
            name: "mcl_trees:leaves_oak".into(),
            drawtype: DrawType::AllfacesOptional,
            paramtype: "light".into(),
            ..Default::default()
        });
        let torch = reg.register_node(NodeDef {
            name: "mcl_torches:torch".into(),
            drawtype: DrawType::Torchlike,
            sunlight_propagates: true,
            light_source: 14,
            ..Default::default()
        });

        let tmp = std::env::temp_dir().join("cubex_light_relight_test");
        let world = WorldStore::new(&tmp, 9);
        let mut ch = Chunk::new(IVec3::ZERO);
        // Sealed roof at y=6
        for z in 0..16 {
            for x in 0..16 {
                ch.set(x, 6, z, stone, 0, 0);
            }
        }
        ch.set(8, 7, 8, leaves, 0, 0);
        propagate_chunk_light(&mut ch, &reg, true);
        // Leaves receive sky BFS > 0 (not 0!) and interior under roof is dark (sky == 0)
        assert!(get_sky_light(ch.get(8, 7, 8).1) >= 14);
        assert_eq!(get_sky_light(ch.get(8, 3, 8).1), 0);
        world.insert_chunk(ch);

        // Place torch inside dark room -> block light floods neighbors
        world.set_node(IVec3::new(8, 3, 8), torch, 0, 0);
        let touched = relight_around_pos(&world, IVec3::new(8, 3, 8), &reg);
        assert!(!touched.is_empty());
        assert_eq!(get_block_light(world.get_node(IVec3::new(9, 3, 8)).1), 13);

        // Break torch -> block light unlights back to 0
        world.set_node(IVec3::new(8, 3, 8), CONTENT_AIR, 0, 0);
        relight_around_pos(&world, IVec3::new(8, 3, 8), &reg);
        assert_eq!(get_block_light(world.get_node(IVec3::new(9, 3, 8)).1), 0);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn multichunk_column_skylight_and_leaf_attenuation() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let leaves = reg.register_node(NodeDef {
            name: "mcl_trees:leaves_oak".into(),
            drawtype: DrawType::AllfacesOptional,
            paramtype: "light".into(),
            ..Default::default()
        });

        let tmp = std::env::temp_dir().join("cubex_light_column_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let world = WorldStore::new(&tmp, 11);
        // Simulate chunks cy = 3, 4, 5 (ground at Y=64 in cy=4, canopy spanning cy=4..5)
        let mut c3 = Chunk::new(IVec3::new(0, 3, 0));
        let mut c4 = Chunk::new(IVec3::new(0, 4, 0));
        let c5 = Chunk::new(IVec3::new(0, 5, 0));
        for z in 0..16 {
            for x in 0..16 {
                for y in 0..16 {
                    c3.set(x, y, z, stone, 0, 0);
                }
                c4.set(x, 0, z, stone, 0, 0);
            }
        }
        // 3x3x2 canopy at (7..=9, 70..=71, 7..=9) inside cy=4 (top=71 -> ly=7, bottom=70 -> ly=6)
        for dz in 7..=9 {
            for dx in 7..=9 {
                c4.set(dx, 7, dz, leaves, 0, 0);
                c4.set(dx, 6, dz, leaves, 0, 0);
            }
        }
        world.insert_chunk(c3);
        world.insert_chunk(c4);
        world.insert_chunk(c5);

        let _ = relight_world_column(&world, 0, 0, &reg);

        // Open air at Y=66 inside cy=4 receives sky = 15 down from cy=5
        assert_eq!(get_sky_light(world.get_node(IVec3::new(2, 66, 2)).1), 15);
        // Interior canopy leaves attenuate gently (15 -> 14 -> 13), never dropping to 0
        assert_eq!(get_sky_light(world.get_node(IVec3::new(8, 71, 8)).1), 14);
        assert_eq!(get_sky_light(world.get_node(IVec3::new(8, 70, 8)).1), 13);
        assert!(get_sky_light(world.get_node(IVec3::new(8, 69, 8)).1) >= 13);
        // Underground stone inside cy=3 stays sky = 0
        assert_eq!(get_sky_light(world.get_node(IVec3::new(8, 55, 8)).1), 0);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
