# Cubex (`cubex-rs`) Architecture

`cubex-rs` is a native Linux Rust voxel engine (`winit` + `wgpu` + `egui` + `rayon`) embedding LuaJIT (`mlua` with `luajit` + `vendored`) to load and run the **Mineclonia** Lua game tree, **Craft & Ruin 32x** resource packs, and `cubex_bridge` Lua overrides on Vulkan hardware.

---

## 1. Subsystem Overview

| Rust Module | Responsibility | Mineclonia / Luanti API Mapping |
| --- | --- | --- |
| `src/paths.rs` | Resolves Mineclonia game directory, Luanti `builtin/`, resource packs, worlds, `minetest.conf`, `cubex_bridge`, `settings.toml`, and `playit` binary | `core.get_modpath`, `core.get_builtin_path`, `core.get_worldpath` |
| `src/config.rs` | Deduplicating `minetest.conf` reader/writer and `EngineSettings` persisted in `settings.toml` (`Performance` / `Balanced` / `Cinematic` / `Custom` presets, shadows, bloom, audio, controls, gamerules) | `core.settings` (`Settings` userdata), `core.get_mapgen_setting`, `core.set_mapgen_setting` |
| `src/registry.rs` | `NodeRegistry`, `ItemRegistry`, `CraftRegistry`, `BiomeRegistry`, `OreRegistry`, `DecorationRegistry`, `EntityRegistry` | `core.register_node`, `core.register_craftitem`, `core.register_tool`, `core.register_alias`, `core.register_craft`, `core.register_biome`, `core.register_ore`, `core.register_decoration`, `core.register_entity`, `core.override_item`, `core.get_content_id`, `core.get_name_from_content_id` |
| `src/world/` | `16x16x16` cubic mapblocks (`nodes: [u16; 4096]`, `param1`, `param2`), BFS skylight/blocklight, multi-biome mapgen (`biome.rs`, `mapgen.rs`), `.mts` schematics, `VoxelManip` bridge, compressed binary persistence | `core.get_node`, `core.set_node`, `core.swap_node`, `core.remove_node`, `core.find_nodes_in_area`, `core.get_voxel_manip`, `VoxelArea`, `core.register_on_generated`, `core.get_node_light` |
| `src/lua_api/` | Bootstraps Luanti `builtin/init.lua`, implements Rust C-API primitives & userdata (`ItemStack`, `VoxelManip`, `Settings`, `ObjectRef`, `InvRef`, `MetaDataRef`, `NodeTimerRef`, `PcgRandom`, `PerlinNoise`, `AreaStore`, `Raycast`), sorts `mod.conf` / `modpack.conf` mods, runs entity/falling-node steps | `core.*` / `minetest.*` global APIs, `core.register_globalstep`, `core.register_abm`, `core.after`, `core.show_formspec`, `core.add_entity` |
| `src/textures/` | Scans mod `textures/` + active resource pack, evaluates Luanti modifiers (`^[combine`, `^[colorize`, `^[multiply`, `^[transform`, `^[mask`, `^[sheet`, `^[verticalframe`, `^[resize`), packs `32x32` layers into a `wgpu` `D2Array` | `tiles = { ... }`, `inventory_image`, `wield_image`, `^[...` texture modifier pipeline |
| `src/render/` | `rayon` parallel greedy meshing (`mesher.rs`), Vulkan renderer (`mod.rs`, `shader.wgsl`), procedural sky & drifting FBM clouds (`sky.rs`, `sky.wgsl`), `2048x2048` directional sun shadow map (`shadow.rs`, `shadow.wgsl`), HDR bloom (`bloom.rs`, `bloom.wgsl`), viewmodel, player/entity rigs (`player_model.rs`, `entity_model.rs`), frustum culling, split-screen & multi-window targets | `drawtype` (`normal`, `allfaces_optional`, `glasslike`, `plantlike`, `torchlike`, `liquid`, `flowingliquid`, `nodebox`, `mesh`), dynamic shadows, sky/clouds, entities |
| `src/physics/` | Swept AABB collision (`collision.rs`) with `0.6`-block stepheight, corner head-wedge guard, eye-center suffocation check, hunger/saturation/exhaustion, natural regen & starvation, food eating (`raycast.rs`), DDA raycast dig/place | `on_dig`, `on_place`, `on_rightclick`, `on_use`, `core.registered_on_dignodes`, `core.registered_on_placenodes` |
| `src/input/` | Gamepad polling via `gilrs` (`gamepad.rs`) and touch button/joystick input handling (`touch.rs`) | Controller & mobile touch bindings |
| `src/audio/` | Non-blocking `.ogg` playback via `rodio` with synthesized UI/pickup/eat waveforms and silent headless fallback | `core.sound_play`, node dig/place sounds |
| `src/ui/` | Native `egui` UI: 9-slice `#727272` stone buttons, `#C6C6C6` inventory panels, splash/loading screens (`splash_screen.rs`, `loading_screen.rs`), rotating panorama title screen (`title_screen.rs`), `ESC` Pause & Settings (`pause_menu.rs`), HUD + F3 overlay (`hud.rs`), death screen (`death_screen.rs`), touch HUD (`touch_hud.rs`), and `FormspecParser` | `core.show_formspec`, `core.close_formspec`, `core.registered_on_player_receive_fields`, `hud_add` / `hud_change` |
| `src/network/` | Async `tokio` UDP/TCP multiplayer server & client (`--dedicated`), LAN discovery (`lan.rs`), server pinger (`servers.rs`), `§`-MOTD parser (`motd.rs`), `CBX-...` invites (`invite.rs`), cloud/browser stream client on port 8090 (`web_client.rs`), and `playit.gg` tunnel manager (`tunnel.rs`) | `cubex_bridge` WAN integration & multiplayer sync |

---

## 2. Content ID & Voxel Coordinate Conventions

- **Reserved Built-in Content IDs (matching Luanti `constants.lua`):**
  - `CONTENT_UNKNOWN = 125`
  - `CONTENT_AIR = 126`
  - `CONTENT_IGNORE = 127`
  - Normal registered nodes skip `125..=127` (`0..=124`, then `128..`).
- **Mapblock Layout (`16x16x16` = 4,096 voxels):**
  - Local coordinates `(lx, ly, lz) in 0..16` map to flat index `(ly * 256) + (lz * 16) + lx`, aligning 1:1 with Luanti's `VoxelArea` `(z * ystride + y * xstride + x)` order when iterating `(y, z, x)`.
- **GPU Texture Array & Greedy Meshing:**
  - Each resolved `32x32` RGBA tile is assigned a `u16` layer index in a `wgpu::TextureDimension::D2` `2DArray` texture (`FilterMode::Nearest`, `AddressMode::Repeat`).
  - Greedy-merged quads span `W x H` voxels and set UVs `(0..W, 0..H)`; the WGSL fragment shader samples `fract(uv)` within the layer with half-texel inset clamping so merged quads tile cleanly without atlas bleeding.
